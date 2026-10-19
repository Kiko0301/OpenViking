import {
  BracesIcon,
  CableIcon,
  Clock3Icon,
  FileTextIcon,
  Loader2Icon,
  RefreshCwIcon,
  ShieldCheckIcon,
  TagIcon,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'

import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Label } from '#/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '#/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '#/components/ui/tabs'
import { Textarea } from '#/components/ui/textarea'
import { cn } from '#/lib/utils'

import type {
  JsonSchema,
  LiveContextMount,
  McpCatalogTool,
} from '../-lib/model'
import { parseLiveContextMetadata, readLiveContext } from '../-lib/api'
import { formatLiveContextResult } from '../-lib/live-result'
import { resolveSchema, schemaFields, schemaType } from '../-lib/schema'

type SchemaArgumentsFormProps = {
  errors: Record<string, string>
  onChange: (name: string, value: string) => void
  schema: JsonSchema
  values: Record<string, string>
}

export function SchemaArgumentsForm({
  errors,
  onChange,
  schema,
  values,
}: SchemaArgumentsFormProps) {
  const { t } = useTranslation('arranger')
  const fields = schemaFields(schema)

  if (!fields.length) {
    return (
      <div className="rounded-md border border-dashed px-3 py-4 text-sm text-muted-foreground">
        {t('mount.noArguments')}
      </div>
    )
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {fields.map((field) => {
        const inputSchema = resolveSchema(field.schema)
        const type = schemaType(inputSchema)
        const id = `mount-argument-${field.name}`
        const value = values[field.name] || ''
        const error = errors[field.name]
        const wide = type === 'object' || type === 'array'

        return (
          <div
            key={field.name}
            className={cn('min-w-0 space-y-1.5', wide && 'sm:col-span-2')}
          >
            <div className="flex min-w-0 items-center gap-2">
              <Label htmlFor={id} className="min-w-0 truncate">
                {field.schema.title || field.name}
              </Label>
              <Badge
                variant={field.required ? 'default' : 'outline'}
                className="shrink-0 px-1.5 py-0 text-[10px]"
              >
                {field.required ? t('fields.required') : t('fields.optional')}
              </Badge>
              <span className="ml-auto font-mono text-[10px] text-muted-foreground">
                {type}
              </span>
            </div>
            {inputSchema.enum ? (
              <Select
                value={value || null}
                onValueChange={(next) =>
                  onChange(
                    field.name,
                    next === null || next === '__unset__' ? '' : String(next),
                  )
                }
              >
                <SelectTrigger id={id} className="w-full">
                  <SelectValue>{value || t('mount.selectValue')}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {!field.required && (
                    <SelectItem value="__unset__">
                      {t('mount.notSet')}
                    </SelectItem>
                  )}
                  {inputSchema.enum.map((option) => (
                    <SelectItem key={String(option)} value={String(option)}>
                      {String(option)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : type === 'boolean' ? (
              <Select
                value={value || null}
                onValueChange={(next) =>
                  onChange(
                    field.name,
                    next === null || next === '__unset__' ? '' : String(next),
                  )
                }
              >
                <SelectTrigger id={id} className="w-full">
                  <SelectValue>{value || t('mount.selectValue')}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {!field.required && (
                    <SelectItem value="__unset__">
                      {t('mount.notSet')}
                    </SelectItem>
                  )}
                  <SelectItem value="true">{t('values.true')}</SelectItem>
                  <SelectItem value="false">{t('values.false')}</SelectItem>
                </SelectContent>
              </Select>
            ) : wide ? (
              <Textarea
                id={id}
                value={value}
                onChange={(event) => onChange(field.name, event.target.value)}
                placeholder={type === 'array' ? '[]' : '{}'}
                className="min-h-24 font-mono text-xs"
                aria-invalid={Boolean(error)}
              />
            ) : (
              <Input
                id={id}
                type={
                  type === 'number' || type === 'integer' ? 'number' : 'text'
                }
                step={
                  type === 'integer' ? 1 : type === 'number' ? 'any' : undefined
                }
                min={inputSchema.minimum}
                max={inputSchema.maximum}
                minLength={inputSchema.minLength}
                maxLength={inputSchema.maxLength}
                value={value}
                onChange={(event) => onChange(field.name, event.target.value)}
                placeholder={
                  inputSchema.examples?.[0] === undefined
                    ? undefined
                    : String(inputSchema.examples[0])
                }
                aria-invalid={Boolean(error)}
              />
            )}
            {(field.schema.description || error) && (
              <p
                className={cn(
                  'text-xs text-muted-foreground',
                  error && 'text-destructive',
                )}
              >
                {error === 'required'
                  ? t('validation.required')
                  : error || field.schema.description}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function LiveContextInspector({
  document,
  mount,
  tool,
}: {
  document: string
  mount: LiveContextMount
  tool?: McpCatalogTool
}) {
  const { t } = useTranslation('arranger')
  const argumentEntries = Object.entries(mount.arguments)
  const annotations = tool?.tool.annotations || {}
  const metadata = parseLiveContextMetadata(document)
  const nodeTitle = metadata?.title || mount.title
  const nodeDescription = metadata?.description || mount.description
  const resultQuery = useQuery({
    enabled: false,
    queryKey: ['arranger-live-result', mount.node_uri],
    queryFn: () => readLiveContext(mount.node_uri),
    retry: false,
  })
  const result =
    resultQuery.data === undefined
      ? null
      : formatLiveContextResult(resultQuery.data)

  return (
    <Tabs defaultValue="result" className="h-full min-h-0 gap-0">
      <TabsList
        variant="line"
        className="h-10 w-full shrink-0 justify-start border-b px-4"
      >
        <TabsTrigger value="result">{t('live.result')}</TabsTrigger>
        <TabsTrigger value="overview">{t('live.overview')}</TabsTrigger>
        <TabsTrigger value="raw">{t('live.rawConfig')}</TabsTrigger>
      </TabsList>
      <TabsContent value="result" className="min-h-0 overflow-y-auto">
        <div className="flex min-h-full flex-col">
          <div className="flex items-center gap-3 border-b p-4">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">{t('live.result')}</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {resultQuery.dataUpdatedAt
                  ? t('live.updatedAt', {
                      value: new Date(
                        resultQuery.dataUpdatedAt,
                      ).toLocaleTimeString(),
                    })
                  : t('live.resultDescription')}
              </p>
            </div>
            <Button
              size="sm"
              disabled={resultQuery.isFetching}
              onClick={() => void resultQuery.refetch()}
            >
              {resultQuery.isFetching ? (
                <Loader2Icon className="animate-spin" />
              ) : (
                <RefreshCwIcon />
              )}
              {result ? t('live.refreshResult') : t('live.loadResult')}
            </Button>
          </div>
          {resultQuery.error ? (
            <div className="p-4 text-sm leading-6 text-destructive">
              {resultQuery.error instanceof Error
                ? resultQuery.error.message
                : String(resultQuery.error)}
            </div>
          ) : result ? (
            <pre
              className={cn(
                'min-h-0 flex-1 whitespace-pre-wrap break-words p-4 font-mono text-xs leading-5',
                result.isError && 'text-destructive',
              )}
            >
              {result.text}
            </pre>
          ) : (
            <div className="flex min-h-48 flex-1 items-center justify-center p-6 text-center text-sm text-muted-foreground">
              {t('live.resultEmpty')}
            </div>
          )}
        </div>
      </TabsContent>
      <TabsContent value="overview" className="min-h-0 overflow-y-auto">
        <div className="divide-y">
          {(nodeTitle || nodeDescription) && (
            <section className="space-y-3 p-4">
              <InspectorHeading
                icon={FileTextIcon}
                label={t('live.nodeInfo')}
              />
              <div>
                {nodeTitle && (
                  <p className="text-sm font-medium">{nodeTitle}</p>
                )}
                {nodeDescription && (
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">
                    {nodeDescription}
                  </p>
                )}
              </div>
            </section>
          )}

          <section className="space-y-3 p-4">
            <InspectorHeading icon={CableIcon} label={t('live.boundTool')} />
            <div className="grid grid-cols-[5rem_minmax(0,1fr)] gap-x-3 gap-y-2 text-xs">
              <span className="text-muted-foreground">
                {t('fields.server')}
              </span>
              <span className="truncate font-mono">{mount.mcp}</span>
              <span className="text-muted-foreground">{t('fields.tool')}</span>
              <span className="truncate font-mono">{mount.tool}</span>
            </div>
          </section>

          <section className="space-y-3 p-4">
            <InspectorHeading icon={BracesIcon} label={t('fields.arguments')} />
            {argumentEntries.length ? (
              <dl className="divide-y rounded-md border">
                {argumentEntries.map(([name, value]) => (
                  <div
                    key={name}
                    className="grid grid-cols-[minmax(5rem,0.35fr)_minmax(0,1fr)] gap-3 px-3 py-2 text-xs"
                  >
                    <dt className="truncate font-mono text-muted-foreground">
                      {name}
                    </dt>
                    <dd className="break-all font-mono">
                      {typeof value === 'string'
                        ? value
                        : JSON.stringify(value)}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-xs text-muted-foreground">
                {t('mount.noArguments')}
              </p>
            )}
          </section>

          <section className="space-y-3 p-4">
            <InspectorHeading icon={Clock3Icon} label={t('live.runtime')} />
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">
                {t('live.timeoutValue', {
                  value: mount.timeout_seconds || 60,
                })}
              </Badge>
              {annotations.readOnlyHint === true && (
                <Badge variant="outline">
                  <ShieldCheckIcon />
                  {t('live.readOnlyHint')}
                </Badge>
              )}
              {annotations.destructiveHint === true && (
                <Badge variant="destructive">{t('live.destructiveHint')}</Badge>
              )}
            </div>
          </section>

          {mount.tags && mount.tags.length > 0 && (
            <section className="space-y-3 p-4">
              <InspectorHeading icon={TagIcon} label={t('fields.tags')} />
              <div className="flex flex-wrap gap-1.5">
                {mount.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </section>
          )}
        </div>
      </TabsContent>
      <TabsContent value="raw" className="min-h-0 overflow-auto">
        <pre className="min-h-full whitespace-pre-wrap break-words p-4 font-mono text-xs leading-5 text-foreground">
          {document}
        </pre>
      </TabsContent>
    </Tabs>
  )
}

function InspectorHeading({
  icon: Icon,
  label,
}: {
  icon: LucideIcon
  label: string
}) {
  return (
    <div className="flex items-center gap-2 text-xs font-semibold">
      <Icon className="size-3.5 text-muted-foreground" />
      <span>{label}</span>
    </div>
  )
}
